update
    Students
set
    Debts = (
        select
            count(distinct CourseId)
        from
            Students S
            left join Plan using (GroupId)
            left join Marks using (StudentId, CourseId)
        where
            S.StudentId = :StudentId
            and Mark is null
    )
where
    StudentId = :StudentId
