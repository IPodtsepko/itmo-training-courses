-- Создание таблиц
create table Groups (
    group_id int,
    group_no char(6)
);
create table Students (
    student_id int,
    name varchar(30),
    group_id int
);

-- Вставка данных
insert into Groups
    (group_id, group_no) values
    (1, 'M34371'),
    (2, 'M34391');
insert into Students
    (student_id, name, group_id) values
    (1, 'Maksim Alzhanov', 2),
    (2, 'Artem Koton', 1),
    (3, 'Anna Suris', 1);

-- Обновление данных (перевод Артема Котона в M3439)
update Students
    set group_id = 2 
    where student_id = 2;
update Students
    set group_id = 2 
    where name = 'Artem Koton';

-- Повторяющийся идентификатор
insert into Groups (group_id, group_no) values
    (1, 'M34381');
delete from Groups where group_no = 'M34381';
alter table Groups
    add constraint group_id_unique unique (group_id);
	
-- Несуществующий идентификатор
update Students set group_id = 5 where student_id = 1;
update Students set group_id = 1 where student_id = 1;
alter table Students add foreign key (group_id)
    references Groups (group_id);
