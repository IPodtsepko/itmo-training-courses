--01.1s
select
    StudentId,
    StudentName,
    GroupId
from
    Students
where
    StudentId = :StudentId


--01.2s
select
    StudentId,
    StudentName,
    GroupId
from
    Students
where
    StudentName = :StudentName


--02.1s
select
    StudentId,
    StudentName,
    GroupName
from
    Students
    natural join Groups
where
    StudentId = :StudentId


--02.2s
select
    StudentId,
    StudentName,
    GroupName
from
    Students
    natural join Groups
where
    StudentName = :StudentName


--03.1s
select
    StudentId,
    StudentName,
    GroupId
from
    Students natural join Marks
where
    Mark = :Mark
    and CourseId = :CourseId


--03.2s
select
    StudentId,
    StudentName,
    GroupId
from
    Marks
    natural join Students
    natural join Courses
where
    Mark = :Mark and CourseName = :CourseName


--03.3s
select
    StudentId,
    StudentName,
    GroupId
from
    Marks
    natural join Students
    natural join Plan
where
    Mark = :Mark and LecturerId = :LecturerId


--03.4s
select
    StudentId,
    StudentName,
    GroupId
from
    Marks
    natural join Students
    natural join Plan
    natural join Lecturers
where
    Mark = :Mark and LecturerName = :LecturerName


--03.5s
select
    StudentId, StudentName, Students.GroupId
from
    Students
    natural join Marks
    inner join Plan on Plan.CourseId = Marks.CourseId
where
    Mark = :Mark and LecturerId = :LecturerId


--03.6s
select
    StudentId, StudentName, Students.GroupId
from
    Students
    natural join Marks
    inner join Plan on Plan.CourseId = Marks.CourseId
    inner join Lecturers on Lecturers.LecturerId = Plan.LecturerId
where
    Mark = :Mark and LecturerName = :LecturerName


--04.1s
select
    StudentId, StudentName, GroupId
from
    Students
except
select
    StudentId, StudentName, GroupId
from
    Students
    natural join Marks
    natural join Courses
where CourseName = :CourseName


--04.2s
select
    StudentId, StudentName, GroupId
from
    Students
    natural join Plan
    natural join Courses
where
    CourseName = :CourseName
except
select
    StudentId, StudentName, GroupId
from
    Students
    natural join Marks
    natural join Courses
where
    CourseName = :CourseName


--05.1s
select
    StudentName, CourseName
from
    (
        select distinct
            StudentId, StudentName, CourseId, CourseName
        from
            Students
            natural join Plan
            natural join Courses
    ) Selected


--05.2s
select
    StudentName, CourseName
from
    (
        select distinct
            Students.StudentId, StudentName, Courses.CourseId, CourseName
        from
            Students
            natural join Plan
            natural join Courses
            left join Marks on Students.StudentId = Marks.StudentId and Courses.CourseId = Marks.CourseId
        where
            Mark is null
    ) Selected


--05.3s
select
    StudentName, CourseName
from
    (
        select distinct
            Students.StudentId, StudentName, Courses.CourseId, CourseName
        from
            Students
            natural join Plan
            natural join Courses
            left join Marks on Students.StudentId = Marks.StudentId and Courses.CourseId = Marks.CourseId
        where
            Mark is null or Mark <> 4 and Mark <> 5
    ) Selected


--06.1s
select distinct
    StudentId
from
    Students
    natural join Marks
    natural join Plan
    natural join Lecturers
where
    LecturerName = :LecturerName


--06.2s
select
    StudentId
from
    Students
except
select
    StudentId
from
    Students
    natural join Marks
    natural join Plan
    natural join Lecturers
where
    LecturerName = :LecturerName


--06.3s
select StudentId from Marks
except
select StudentId
from (
    select
        StudentId, Plan.CourseId
    from
        Marks,
        Plan natural join Lecturers
    where
        LecturerName = :LecturerName
    except
    select StudentId, CourseId from Marks
) S


--06.4s
select StudentId
from (
    select distinct
        Marks.StudentId,
        Students.StudentId as SId
    from
        Students natural join Plan natural join Lecturers, Marks
    where
        LecturerName = :LecturerName
    except
    select StudentId, SId
    from (
        select distinct
            Marks.StudentId,
            Plan.CourseId,
            Students.StudentId as SId
        from
            Students natural join Plan natural join Lecturers, Marks
        where
            LecturerName = :LecturerName
        except
        select
            Marks.StudentId,
            Plan.CourseId,
            Students.StudentId as SId
        from
            Students natural join Plan natural join Lecturers
            inner join Marks on Marks.CourseId = Plan.CourseId
        where
            LecturerName = :LecturerName
    ) S1
) S2
where
    StudentId = SId


--07.1s
select CourseId, GroupId from Marks, Students
except
select CourseId, GroupId from (
    select
        CourseId, Students.StudentId, GroupId
    from
        Marks, Students
    except
    select
        CourseId, Students.StudentId, GroupId
    from
        Marks
        inner join Students on Marks.StudentId = Students.StudentId
) S


--07.2s
select GroupName, CourseName from (
    select CourseId, GroupId from Marks, Students
    except
    select CourseId, GroupId from (
        select
            CourseId, Students.StudentId, GroupId
        from
            Marks, Students
        except
        select
            CourseId, Students.StudentId, GroupId
        from
            Marks
            inner join Students on Marks.StudentId = Students.StudentId
    ) S
) R natural join Groups natural join Courses


--08.1s
select
    sum(Mark) as SumMark
from
    Marks
where
    StudentId = :StudentId


--08.2s
select
    StudentName,
    sum(Mark) as SumMark
from
    Students
    left join Marks on Students.StudentId = Marks.StudentId
group by
    Students.StudentId, StudentName


--08.3s
select
    GroupName,
    sum(Mark) as SumMark
from
    Groups
    left join Students on Groups.GroupId = Students.GroupId
    left join Marks on Students.StudentId = Marks.StudentId
group by
    Groups.GroupId, GroupName


--09.1s
select
    avg(cast(Mark as double)) as AvgMark
from
    Marks
where
    StudentId = :StudentId


--09.2s
select
    StudentName,
    avg(cast(Mark as double)) as AvgMark
from
    Students
    left join Marks on Students.StudentId = Marks.StudentId
group by
    Students.StudentId, StudentName


--09.3s
select
    GroupName,
    avg(cast(Mark as double)) as AvgMark
from
    Groups
    left join Students on Groups.GroupId = Students.GroupId
    left join Marks on Students.StudentId = Marks.StudentId
group by
    Groups.GroupId, GroupName


--09.4s
select
    GroupName, AvgAvgMark
from
    Groups
    natural join (
        select
            GroupId,
            avg(AvgMark) as AvgAvgMark
        from (
            select
                Groups.GroupId,
                avg(cast(Mark as double)) as AvgMark
            from
                Groups
                left join Students on Groups.GroupId = Students.GroupId
                left join Marks on Students.StudentId = Marks.StudentId
            group by
                Groups.GroupId, Students.StudentId
        ) StatisticsWithGroupId
        group by
            GroupId
    ) Statistics


--10s
select
    Students.StudentId,
    count(distinct Plan.CourseId) as Total,
    count(distinct case when Mark is not null then Plan.CourseId end) as Passed,
    count(distinct case when Mark is null then Plan.CourseId end) as Failed
from
    Students
    left join Plan on Students.GroupId = Plan.GroupId
    left join Marks on Students.StudentId = Marks.StudentId and Plan.CourseId = Marks.CourseId
group by
    Students.StudentId


--do
